"""Amazon Managed Grafana provisioning on the administrator deploy path.

The workspace is resolved in the CPU cluster's region and never guessed between
candidates -- and created for the site when the region has none; the dashboards
are pushed through the Grafana HTTP API with a service-account token that lives
for one run and never reaches state, logs or the summary. Every AWS call goes
through a recorded fake runner and every HTTP call through a recorded fake
transport -- nothing here reaches the network.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from gpu_fault.admin import grafana as admin_grafana
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    BootstrapRequest,
    BootstrapState,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.grafana import (
    DASHBOARD_FOLDER_UID,
    DATASOURCE_UID,
    GRAFANA_VIEWER_ENV,
    GRAFANA_WORKSPACE_ID_ENV,
    GrafanaSettings,
    HttpResponse,
    dashboard_asset_digests,
    ensure_grafana_dashboards,
    ensure_grafana_workspace,
    grafana_access_lines,
    grafana_environment,
    grafana_installation_resources,
    grafana_request_fields,
    grafana_settings,
    provision_grafana,
)
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)
from tests.admin._bootstrap_support import _cluster

SITE = "site-a"
REGION = "us-east-1"
AMP = "ws-1283ef61"
HYPERPOD_WORKSPACE = {
    "id": "g-5b81a13d97",
    "name": "sagemaker-observability-39e91dd0-amgws",
    "status": "ACTIVE",
    "endpoint": "g-5b81a13d97.grafana-workspace.us-east-1.amazonaws.com",
    "tags": {"SageMaker": "true"},
    "workspaceRoleArn": "arn:aws:iam::123456789012:role/service-role/GraAcc",
}


class Runner:
    """A recorded ``aws`` CLI: answers list/describe from a workspace table."""

    def __init__(
        self,
        workspaces: Sequence[Mapping[str, Any]] = (),
        *,
        service_accounts: Sequence[Mapping[str, Any]] = (),
        create_refused: str | None = None,
        permission_errors: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        # Deep copies: the fake tags workspaces in place, and the fixtures are
        # module constants shared by every test.
        self.workspaces = [json.loads(json.dumps(item)) for item in workspaces]
        self.service_accounts = [dict(item) for item in service_accounts]
        # The region's answer to create-workspace when it will not create one.
        self.create_refused = create_refused
        self.permission_errors = [dict(item) for item in permission_errors]
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.token_key = "glsa_secret_token_value"

    def _find(self, workspace_id: str) -> dict[str, Any] | None:
        return next(
            (item for item in self.workspaces if item["id"] == workspace_id), None
        )

    def run(self, arguments: Sequence[str], **keywords: Any) -> str:
        argv = list(arguments)
        self.calls.append((argv, keywords))
        operation = argv[2] if argv[0] == "aws" else argv[0]
        if argv[1] == "grafana":
            return self._grafana(operation, argv, keywords)
        if argv[1] == "iam":
            return self._iam(operation, argv)
        raise AssertionError(argv)

    def _grafana(self, operation: str, argv: list[str], keywords: Any) -> str:
        if operation == "list-workspaces":
            return json.dumps({"workspaces": self.workspaces})
        if operation == "describe-workspace":
            workspace = self._find(argv[argv.index("--workspace-id") + 1])
            if workspace is None:
                raise BootstrapError(
                    "command failed (254): aws: An error occurred "
                    "(ResourceNotFoundException) when calling DescribeWorkspace"
                )
            return json.dumps({"workspace": workspace})
        if operation == "tag-resource":
            assert keywords.get("mutate") is True, "tagging is a write"
            workspace_id = argv[argv.index("--resource-arn") + 1].rsplit("/", 1)[-1]
            workspace = self._find(workspace_id)
            assert workspace is not None, argv
            key, _sep, value = argv[argv.index("--tags") + 1].partition("=")
            workspace.setdefault("tags", {})[key] = value
            return ""
        if operation == "create-workspace":
            assert keywords.get("mutate") is True, "creation is a write"
            if self.create_refused:
                raise BootstrapError(
                    f"command failed (254): aws: An error occurred "
                    f"(ValidationException) when calling CreateWorkspace: "
                    f"{self.create_refused}"
                )
            tags = json.loads(argv[argv.index("--tags") + 1])
            created = {
                "id": "g-created01",
                "name": argv[argv.index("--workspace-name") + 1],
                "status": "CREATING",
                "endpoint": "g-created01.grafana-workspace.us-east-1.amazonaws.com",
                "tags": tags,
                "workspaceRoleArn": argv[argv.index("--workspace-role-arn") + 1],
            }
            self.workspaces.append(created)
            # The next describe sees it ACTIVE: one poll, no sleep.
            created_view = dict(created)
            created["status"] = "ACTIVE"
            return json.dumps({"workspace": created_view})
        if operation == "list-workspace-service-accounts":
            return json.dumps({"serviceAccounts": self.service_accounts})
        if operation == "create-workspace-service-account":
            assert keywords.get("mutate") is True, "service account creation writes"
            account = {
                "id": "7",
                "name": argv[argv.index("--name") + 1],
                "grafanaRole": argv[argv.index("--grafana-role") + 1],
            }
            self.service_accounts.append(account)
            return json.dumps(account)
        if operation == "create-workspace-service-account-token":
            assert keywords.get("mutate") is True, "token creation writes"
            assert keywords.get("sensitive") is True, "the token command is sensitive"
            return json.dumps(
                {
                    "serviceAccountToken": {
                        "id": "token-1",
                        "name": argv[argv.index("--name") + 1],
                        "key": self.token_key,
                    },
                    "serviceAccountId": argv[argv.index("--service-account-id") + 1],
                }
            )
        if operation == "delete-workspace-service-account-token":
            assert keywords.get("mutate") is True, "token deletion writes"
            return ""
        if operation == "update-permissions":
            assert keywords.get("mutate") is True, "granting a role writes"
            return json.dumps({"errors": self.permission_errors})
        raise AssertionError(argv)

    def _iam(self, operation: str, argv: list[str]) -> str:
        if operation == "get-role":
            raise BootstrapError(
                "command failed (254): aws: An error occurred (NoSuchEntity)"
            )
        if operation in {"create-role", "put-role-policy"}:
            if operation == "create-role":
                name = argv[argv.index("--role-name") + 1]
                return json.dumps(
                    {"Role": {"Arn": f"arn:aws:iam::123456789012:role/{name}"}}
                )
            return ""
        raise AssertionError(argv)

    def aws_json(self, region: str, *arguments: str, **keywords: Any) -> Any:
        assert region == REGION, f"Grafana followed region {region}, not {REGION}"
        return json.loads(self.run(["aws", *arguments, "--region", region], **keywords))

    def operations(self) -> list[str]:
        return [argv[2] for argv, _keywords in self.calls if argv[0] == "aws"]


class Http:
    """A recorded Grafana HTTP transport answering from a (method, path) table."""

    def __init__(self, responses: Mapping[tuple[str, str], Any] | None = None) -> None:
        self.responses = dict(responses or {})
        self.requests: list[tuple[str, str, dict[str, str], Any]] = []

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> HttpResponse:
        path = url.split("amazonaws.com", 1)[1]
        payload = json.loads(body) if body else None
        self.requests.append((method, path, dict(headers), payload))
        answer = self.responses.get((method, path))
        if answer is None:
            if method == "POST" and path == "/api/dashboards/db":
                return HttpResponse(
                    200,
                    json.dumps(
                        {
                            "uid": payload["dashboard"]["uid"],
                            "version": 3,
                            "status": "success",
                            "url": f"/d/{payload['dashboard']['uid']}/x",
                        }
                    ),
                )
            return HttpResponse(200, json.dumps({"status": "OK"}))
        if isinstance(answer, HttpResponse):
            return answer
        return HttpResponse(200, json.dumps(answer))

    def paths(self, method: str | None = None) -> list[str]:
        return [
            path
            for verb, path, _headers, _payload in self.requests
            if method is None or verb == method
        ]


def _dashboards(tmp_path: Path, *uids: str) -> Path:
    directory = tmp_path / "deploy/observability/dashboards"
    directory.mkdir(parents=True, exist_ok=True)
    for uid in uids:
        (directory / f"{uid}.json").write_text(
            json.dumps(
                {
                    "uid": uid,
                    "title": f"GPU Fault {uid}",
                    "panels": [{"datasource": {"uid": DATASOURCE_UID}}],
                }
            ),
            encoding="utf-8",
        )
    return directory


def _tagged(workspace: Mapping[str, Any], **tags: str) -> dict[str, Any]:
    return {**workspace, "tags": {**workspace.get("tags", {}), **tags}}


# --- workspace resolution -----------------------------------------------------


def test_the_requested_workspace_is_external_when_it_carries_no_site_tag() -> None:
    runner = Runner([HYPERPOD_WORKSPACE])

    workspace = ensure_grafana_workspace(
        runner, cpu=_cluster(), site_id=SITE, requested_id="g-5b81a13d97"
    )

    assert workspace["workspace_id"] == "g-5b81a13d97"
    assert workspace["ownership"] == "EXTERNAL"
    assert workspace["endpoint"] == HYPERPOD_WORKSPACE["endpoint"]
    assert workspace["role_arn"] == HYPERPOD_WORKSPACE["workspaceRoleArn"]
    assert "create-workspace" not in runner.operations()


def test_the_requested_workspace_is_reused_when_tagged_for_this_site() -> None:
    runner = Runner([_tagged(HYPERPOD_WORKSPACE, **{"gpu-fault:site-id": SITE})])

    workspace = ensure_grafana_workspace(
        runner, cpu=_cluster(), site_id=SITE, requested_id="g-5b81a13d97"
    )

    assert workspace["ownership"] == "REUSED"


def test_the_requested_workspace_is_created_when_it_carries_our_creation_tag() -> None:
    tags = {
        "gpu-fault:site-id": SITE,
        "gpu-fault:grafana-created-by": "gpu-fault-admin",
    }
    runner = Runner([_tagged(HYPERPOD_WORKSPACE, **tags)])

    workspace = ensure_grafana_workspace(
        runner, cpu=_cluster(), site_id=SITE, requested_id="g-5b81a13d97"
    )

    assert workspace["ownership"] == "CREATED"


def test_a_requested_workspace_that_does_not_exist_is_an_error() -> None:
    runner = Runner([HYPERPOD_WORKSPACE])

    with pytest.raises(BootstrapError, match="g-missing"):
        ensure_grafana_workspace(
            runner, cpu=_cluster(), site_id=SITE, requested_id="g-missing"
        )


def test_a_requested_workspace_of_another_site_is_refused() -> None:
    runner = Runner([_tagged(HYPERPOD_WORKSPACE, **{"gpu-fault:site-id": "other"})])

    with pytest.raises(BootstrapError, match="other"):
        ensure_grafana_workspace(
            runner, cpu=_cluster(), site_id=SITE, requested_id="g-5b81a13d97"
        )


def test_a_requested_workspace_that_is_not_active_is_an_error() -> None:
    runner = Runner([{**HYPERPOD_WORKSPACE, "status": "UPDATING"}])

    with pytest.raises(BootstrapError, match="UPDATING"):
        ensure_grafana_workspace(
            runner, cpu=_cluster(), site_id=SITE, requested_id="g-5b81a13d97"
        )


def test_a_workspace_tagged_for_this_site_is_reused_without_writing() -> None:
    other = {**HYPERPOD_WORKSPACE, "id": "g-other0001", "name": "other"}
    runner = Runner([other, _tagged(HYPERPOD_WORKSPACE, **{"gpu-fault:site-id": SITE})])

    workspace = ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)

    assert workspace["workspace_id"] == "g-5b81a13d97"
    assert workspace["ownership"] == "REUSED"
    assert not any(keywords.get("mutate") for _argv, keywords in runner.calls), (
        "resolving a tagged workspace wrote to AWS"
    )


def test_the_only_active_workspace_is_adopted_and_tagged_for_the_next_run() -> None:
    runner = Runner(
        [
            HYPERPOD_WORKSPACE,
            {**HYPERPOD_WORKSPACE, "id": "g-deleting", "status": "DELETING"},
        ]
    )

    workspace = ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)

    assert workspace["workspace_id"] == "g-5b81a13d97"
    assert workspace["ownership"] == "EXTERNAL"
    tag_call = next(argv for argv, _k in runner.calls if argv[2] == "tag-resource")
    assert tag_call[tag_call.index("--tags") + 1] == f"gpu-fault:site-id={SITE}"
    assert tag_call[tag_call.index("--resource-arn") + 1].endswith(
        ":123456789012:/workspaces/g-5b81a13d97"
    ), "the tag went to a resource ARN that is not the workspace's"
    # The adopted workspace resolves by tag on the next run, still not ours.
    again = ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)
    assert again["ownership"] == "REUSED"


def test_two_untagged_active_workspaces_are_never_guessed_between() -> None:
    second = {**HYPERPOD_WORKSPACE, "id": "g-second0002", "name": "team-grafana"}
    runner = Runner([HYPERPOD_WORKSPACE, second])

    with pytest.raises(BootstrapError) as failure:
        ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)

    message = str(failure.value)
    assert "g-5b81a13d97" in message and "g-second0002" in message
    assert "--grafana-workspace-id" in message
    assert "tag-resource" not in runner.operations()


def test_resolution_order_is_operator_id_then_site_tag_then_single_active_then_create() -> (
    None
):
    """Each step of the order wins over the ones below it, and no step guesses."""

    ours = _tagged(
        {**HYPERPOD_WORKSPACE, "id": "g-ours000001", "name": "ours"},
        **{"gpu-fault:site-id": SITE},
    )
    # 1. The operator's id wins even when a tagged workspace exists.
    runner = Runner([HYPERPOD_WORKSPACE, ours])
    chosen = ensure_grafana_workspace(
        runner, cpu=_cluster(), site_id=SITE, requested_id="g-5b81a13d97"
    )
    assert chosen["workspace_id"] == "g-5b81a13d97"
    # 2. The site tag wins over an untagged ACTIVE workspace.
    chosen = ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)
    assert chosen["workspace_id"] == "g-ours000001"
    # 3. The region's single ACTIVE workspace is adopted.
    runner = Runner([HYPERPOD_WORKSPACE])
    chosen = ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)
    assert chosen["workspace_id"] == "g-5b81a13d97"
    assert "create-workspace" not in runner.operations(), (
        "a workspace was created although the region had one to adopt"
    )
    # 4. An empty region gets a workspace created for the site.
    runner = Runner([])
    chosen = ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)
    assert chosen["workspace_id"] == "g-created01"
    assert chosen["ownership"] == "CREATED"


def test_an_empty_region_gets_a_workspace_with_sso_customer_managed_and_our_role() -> (
    None
):
    runner = Runner([])

    workspace = ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)

    assert workspace["ownership"] == "CREATED"
    assert workspace["workspace_id"] == "g-created01"
    assert workspace["role_ownership"] == "CREATED"
    assert workspace["role_name"] == "gpu-fault-site-a-grafana"
    create = next(argv for argv, _k in runner.calls if argv[2] == "create-workspace")
    assert create[create.index("--authentication-providers") + 1] == "AWS_SSO"
    assert create[create.index("--permission-type") + 1] == "CUSTOMER_MANAGED"
    assert create[create.index("--account-access-type") + 1] == "CURRENT_ACCOUNT"
    assert create[create.index("--workspace-role-arn") + 1].endswith(
        "role/gpu-fault-site-a-grafana"
    ), "the workspace was created with a role other than the one ensured for it"
    assert json.loads(create[create.index("--tags") + 1]) == {
        "gpu-fault:site-id": SITE,
        "gpu-fault:grafana-created-by": "gpu-fault-admin",
    }
    policy_call = next(
        argv for argv, _k in runner.calls if argv[2] == "put-role-policy"
    )
    document = json.loads(policy_call[policy_call.index("--policy-document") + 1])
    actions = set(document["Statement"][0]["Action"])
    assert {
        "aps:QueryMetrics",
        "aps:GetSeries",
        "aps:GetLabels",
        "aps:GetMetricMetadata",
        "aps:ListRules",
        "aps:ListAlertManagerAlerts",
    } <= actions
    trust_call = next(argv for argv, _k in runner.calls if argv[2] == "create-role")
    trust = json.loads(
        trust_call[trust_call.index("--assume-role-policy-document") + 1]
    )
    assert trust["Statement"][0]["Principal"] == {"Service": "grafana.amazonaws.com"}


def test_a_region_that_refuses_the_creation_names_the_manual_fallback() -> None:
    runner = Runner([], create_refused="No IAM Identity Center instance in region")

    with pytest.raises(BootstrapError) as failure:
        ensure_grafana_workspace(runner, cpu=_cluster(), site_id=SITE)

    message = str(failure.value)
    assert "No IAM Identity Center instance" in message
    assert "--grafana-workspace-id" in message, "the fallback flag is not named"
    assert "console" in message.lower(), "the manual creation path is not named"


def test_a_read_only_probe_of_an_untagged_workspace_reports_the_write() -> None:
    runner = Runner([HYPERPOD_WORKSPACE])

    with pytest.raises(BootstrapMutationRequired):
        ensure_grafana_workspace(
            ReadOnlyProbeRunner(runner), cpu=_cluster(), site_id=SITE
        )


# --- provisioning over HTTP ---------------------------------------------------


def _workspace(**overrides: Any) -> dict[str, Any]:
    return {
        "workspace_id": "g-5b81a13d97",
        "endpoint": HYPERPOD_WORKSPACE["endpoint"],
        "ownership": "EXTERNAL",
        "region": REGION,
        **overrides,
    }


def test_first_provisioning_creates_datasource_folder_and_dashboards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = Runner([HYPERPOD_WORKSPACE])
    http = Http(
        {
            ("GET", f"/api/datasources/uid/{DATASOURCE_UID}"): HttpResponse(
                404, '{"message":"Data source not found"}'
            ),
            ("GET", f"/api/folders/{DASHBOARD_FOLDER_UID}"): HttpResponse(
                404, '{"message":"folder not found"}'
            ),
        }
    )

    summary = provision_grafana(
        runner,
        workspace=_workspace(),
        amp_workspace_id=AMP,
        region=REGION,
        dashboards_dir=_dashboards(tmp_path, "gpu-fault-overview", "gpu-fault-nodes"),
        http=http,
    )

    assert http.paths() == [
        "/api/org",
        f"/api/datasources/uid/{DATASOURCE_UID}",
        "/api/datasources",
        f"/api/datasources/uid/{DATASOURCE_UID}/health",
        f"/api/folders/{DASHBOARD_FOLDER_UID}",
        "/api/folders",
        "/api/dashboards/db",
        "/api/dashboards/db",
    ]
    _method, _path, headers, datasource = http.requests[2]
    assert headers["Authorization"] == f"Bearer {runner.token_key}"
    assert datasource == {
        "uid": DATASOURCE_UID,
        "name": f"GPU Fault AMP ({REGION})",
        "type": "prometheus",
        "access": "proxy",
        "url": f"https://aps-workspaces.{REGION}.amazonaws.com/workspaces/{AMP}",
        "isDefault": False,
        "jsonData": {
            "httpMethod": "POST",
            "sigV4Auth": True,
            "sigV4AuthType": "default",
            "sigV4Region": REGION,
            "manageAlerts": True,
            "prometheusType": "Prometheus",
        },
    }
    folder = http.requests[5][3]
    assert folder == {"uid": DASHBOARD_FOLDER_UID, "title": "GPU Fault Recovery"}
    imports = [
        payload
        for verb, path, _h, payload in http.requests
        if path == "/api/dashboards/db"
    ]
    assert [item["dashboard"]["uid"] for item in imports] == [
        "gpu-fault-nodes",
        "gpu-fault-overview",
    ]
    assert all(item["overwrite"] is True for item in imports), "imports must overwrite"
    assert all(item["folderUid"] == DASHBOARD_FOLDER_UID for item in imports), (
        "a dashboard was imported outside the solution folder"
    )
    assert all(item["dashboard"].get("id") is None for item in imports), (
        "a dashboard carried a numeric id, which binds it to one Grafana instance"
    )
    assert summary["datasource_uid"] == DATASOURCE_UID
    assert summary["folder_uid"] == DASHBOARD_FOLDER_UID
    assert summary["dashboards"] == [
        {"uid": "gpu-fault-nodes", "title": "GPU Fault gpu-fault-nodes", "version": 3},
        {
            "uid": "gpu-fault-overview",
            "title": "GPU Fault gpu-fault-overview",
            "version": 3,
        },
    ]
    assert summary["workspace_url"] == f"https://{HYPERPOD_WORKSPACE['endpoint']}"
    assert summary["dashboards_url"] == (
        f"https://{HYPERPOD_WORKSPACE['endpoint']}/dashboards/f/{DASHBOARD_FOLDER_UID}"
    )
    operations = runner.operations()
    assert operations.index("create-workspace-service-account") < operations.index(
        "create-workspace-service-account-token"
    )
    assert operations[-1] == "delete-workspace-service-account-token", (
        "the short-lived token was not deleted after the run"
    )
    token_create = next(
        argv
        for argv, _k in runner.calls
        if argv[2] == "create-workspace-service-account-token"
    )
    assert token_create[token_create.index("--seconds-to-live") + 1] == "900"
    assert token_create[token_create.index("--name") + 1].startswith(
        "gpu-fault-provisioner-"
    ), "the one-run token is not named after the provisioner"
    assert runner.token_key not in json.dumps(summary), "the token leaked into state"
    assert runner.token_key not in capsys.readouterr().err, "the token was printed"


def test_a_second_run_updates_in_place_and_reuses_the_service_account(
    tmp_path: Path,
) -> None:
    runner = Runner(
        [HYPERPOD_WORKSPACE],
        service_accounts=[
            {"id": "3", "name": "SageMakerObservability", "grafanaRole": "ADMIN"},
            {"id": "9", "name": "gpu-fault-provisioner", "grafanaRole": "ADMIN"},
        ],
    )
    http = Http(
        {
            ("GET", f"/api/datasources/uid/{DATASOURCE_UID}"): {
                "id": 12,
                "uid": DATASOURCE_UID,
                "name": "old name",
            },
            ("GET", f"/api/folders/{DASHBOARD_FOLDER_UID}"): {
                "uid": DASHBOARD_FOLDER_UID,
                "title": "GPU Fault Recovery",
            },
        }
    )

    provision_grafana(
        runner,
        workspace=_workspace(),
        amp_workspace_id=AMP,
        region=REGION,
        dashboards_dir=_dashboards(tmp_path, "gpu-fault-overview"),
        http=http,
    )

    assert http.paths("PUT") == [f"/api/datasources/uid/{DATASOURCE_UID}"]
    assert "/api/datasources" not in http.paths("POST")
    assert "/api/folders" not in http.paths("POST")
    assert "create-workspace-service-account" not in runner.operations()
    token_create = next(
        argv
        for argv, _k in runner.calls
        if argv[2] == "create-workspace-service-account-token"
    )
    assert token_create[token_create.index("--service-account-id") + 1] == "9", (
        "the provisioner token was minted for another service account"
    )


def test_the_token_is_deleted_when_grafana_refuses_and_the_error_is_specific(
    tmp_path: Path,
) -> None:
    runner = Runner([HYPERPOD_WORKSPACE])
    http = Http(
        {("GET", "/api/org"): HttpResponse(503, "database is locked " + "x" * 400)}
    )

    with pytest.raises(BootstrapError) as failure:
        provision_grafana(
            runner,
            workspace=_workspace(),
            amp_workspace_id=AMP,
            region=REGION,
            dashboards_dir=_dashboards(tmp_path, "gpu-fault-overview"),
            http=http,
        )

    message = str(failure.value)
    assert "503" in message and "/api/org" in message
    assert "database is locked" in message
    assert len(message) < 500, "the error carried the whole response body"
    assert runner.token_key not in message
    assert runner.operations()[-1] == "delete-workspace-service-account-token"


def test_datasource_verification_falls_back_to_a_query_and_then_fails(
    tmp_path: Path,
) -> None:
    unhealthy = HttpResponse(400, '{"status":"ERROR","message":"403 Forbidden"}')
    responses = {("GET", f"/api/datasources/uid/{DATASOURCE_UID}/health"): unhealthy}
    http = Http(responses)

    provision_grafana(
        Runner([HYPERPOD_WORKSPACE]),
        workspace=_workspace(),
        amp_workspace_id=AMP,
        region=REGION,
        dashboards_dir=tmp_path / "absent",
        http=http,
    )

    query = next(
        payload for verb, path, _h, payload in http.requests if path == "/api/ds/query"
    )
    assert query["queries"][0]["datasource"] == {"uid": DATASOURCE_UID}
    assert query["queries"][0]["expr"] == "up"

    http = Http({**responses, ("POST", "/api/ds/query"): HttpResponse(400, "no sigv4")})
    with pytest.raises(BootstrapError, match="gpu-fault-amp.*403 Forbidden"):
        provision_grafana(
            Runner([HYPERPOD_WORKSPACE]),
            workspace=_workspace(),
            amp_workspace_id=AMP,
            region=REGION,
            dashboards_dir=tmp_path / "absent",
            http=http,
        )


def test_a_missing_dashboards_directory_imports_nothing_and_still_succeeds(
    tmp_path: Path,
) -> None:
    http = Http()

    summary = provision_grafana(
        Runner([HYPERPOD_WORKSPACE]),
        workspace=_workspace(),
        amp_workspace_id=AMP,
        region=REGION,
        dashboards_dir=tmp_path / "does-not-exist",
        http=http,
    )

    assert summary["dashboards"] == []
    assert "/api/dashboards/db" not in http.paths()


def test_a_dashboard_without_a_stable_uid_is_rejected_before_any_call(
    tmp_path: Path,
) -> None:
    directory = _dashboards(tmp_path, "gpu-fault-overview")
    (directory / "broken.json").write_text('{"title": "no uid"}', encoding="utf-8")
    runner = Runner([HYPERPOD_WORKSPACE])
    http = Http()

    with pytest.raises(BootstrapError, match="broken.json"):
        provision_grafana(
            runner,
            workspace=_workspace(),
            amp_workspace_id=AMP,
            region=REGION,
            dashboards_dir=directory,
            http=http,
        )

    assert http.requests == [] and runner.calls == [], (
        "a broken asset reached the network before validation"
    )


# --- the monitoring task step ----------------------------------------------------


def _settings(**overrides: Any) -> GrafanaSettings:
    return replace(GrafanaSettings(), **overrides)


def _ensure(
    runner: Any,
    settings: GrafanaSettings | None,
    tmp_path: Path,
    *,
    probe_only: bool = False,
    http: Http | None = None,
) -> dict[str, Any]:
    return ensure_grafana_dashboards(
        runner,
        settings=settings,
        cpu=_cluster(),
        site_id=SITE,
        amp_workspace_id=AMP,
        repository_root=tmp_path,
        probe_only=probe_only,
        http=http or Http(),
    )


def test_a_caller_without_a_grafana_decision_skips_the_step(tmp_path: Path) -> None:
    runner = Runner([HYPERPOD_WORKSPACE])

    assert _ensure(runner, None, tmp_path) == {"status": "SKIPPED"}
    assert runner.calls == []


def test_the_default_settings_create_and_provision_when_the_region_is_empty(
    tmp_path: Path,
) -> None:
    """First deploy for the customer: no flag, no workspace, and the dashboards
    still land -- in a workspace recorded as ours."""

    runner = Runner([])
    _dashboards(tmp_path, "gpu-fault-overview")

    result = _ensure(runner, _settings(), tmp_path)

    assert result["status"] == "PROVISIONED"
    assert result["workspace_id"] == "g-created01"
    assert result["ownership"] == "CREATED"
    assert [item["uid"] for item in result["dashboards"]] == ["gpu-fault-overview"]


def test_a_refused_creation_degrades_to_a_warning_and_the_deploy_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = Runner([], create_refused="No IAM Identity Center instance in region")

    result = _ensure(runner, _settings(), tmp_path)

    assert result["status"] == "FAILED"
    assert "No IAM Identity Center instance" in result["reason"]
    err = capsys.readouterr().err
    assert "WARNING" in err and "deploy continues" in err
    assert "--grafana-workspace-id" in err, "the warning does not name the fallback"
    assert "create-workspace-service-account-token" not in runner.operations()


def test_the_viewer_flag_grants_the_identity_center_user_after_the_import(
    tmp_path: Path,
) -> None:
    runner = Runner([HYPERPOD_WORKSPACE])

    result = _ensure(runner, _settings(viewer_sso_user_id="u-42"), tmp_path)

    assert result["status"] == "PROVISIONED"
    assert result["viewer_sso_user_id"] == "u-42"
    operations = runner.operations()
    assert operations.index("update-permissions") > operations.index(
        "delete-workspace-service-account-token"
    ), "the grant ran before the import finished"
    grant = next(argv for argv, _k in runner.calls if argv[2] == "update-permissions")
    assert grant[grant.index("--workspace-id") + 1] == "g-5b81a13d97"
    batch = json.loads(grant[grant.index("--update-instruction-batch") + 1])
    assert batch == [
        {
            "action": "ADD",
            "role": "VIEWER",
            "users": [{"id": "u-42", "type": "SSO_USER"}],
        }
    ]


def test_a_refused_viewer_is_operator_input_and_fails_the_deploy(
    tmp_path: Path,
) -> None:
    runner = Runner(
        [HYPERPOD_WORKSPACE],
        permission_errors=[{"code": 1, "message": "user u-typo not found"}],
    )

    with pytest.raises(BootstrapError, match="u-typo"):
        _ensure(runner, _settings(viewer_sso_user_id="u-typo"), tmp_path)


def test_the_access_lines_name_the_real_workspace_and_the_exact_permission_command(
    tmp_path: Path,
) -> None:
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id=SITE)
    assert grafana_access_lines(state) == [], "nothing provisioned, nothing to say"
    state.record(
        "monitoring_install",
        {
            "grafana": {
                "status": "PROVISIONED",
                "workspace_id": "g-5b81a13d97",
                "region": REGION,
                "dashboards_url": "https://g-5b81a13d97.grafana-workspace/x",
            }
        },
    )

    lines = grafana_access_lines(state)

    assert lines[0] == "Grafana dashboards: https://g-5b81a13d97.grafana-workspace/x"
    assert (
        "aws grafana update-permissions --region us-east-1 "
        "--workspace-id g-5b81a13d97 --update-instruction-batch "
        '\'[{"action":"ADD","role":"VIEWER","users":'
        '[{"id":"<sso-user-id>","type":"SSO_USER"}]}]\''
    ) in lines[1]
    assert "--grafana-viewer" in lines[1]

    state.record(
        "monitoring_install",
        {
            "grafana": {
                "status": "PROVISIONED",
                "workspace_id": "g-5b81a13d97",
                "region": REGION,
                "dashboards_url": "https://g-5b81a13d97.grafana-workspace/x",
                "viewer_sso_user_id": "u-42",
            }
        },
    )
    granted = grafana_access_lines(state)
    assert "u-42" in granted[1] and "update-permissions" not in granted[1]


def test_a_provisioned_step_records_the_workspace_and_the_imports_not_the_token(
    tmp_path: Path,
) -> None:
    runner = Runner([HYPERPOD_WORKSPACE])
    _dashboards(tmp_path, "gpu-fault-overview")

    result = _ensure(runner, _settings(), tmp_path)

    assert result["status"] == "PROVISIONED"
    assert result["workspace_id"] == "g-5b81a13d97"
    assert result["ownership"] == "EXTERNAL"
    assert result["endpoint"] == HYPERPOD_WORKSPACE["endpoint"]
    assert [item["uid"] for item in result["dashboards"]] == ["gpu-fault-overview"]
    assert runner.token_key not in json.dumps(result)


def test_an_ambiguous_region_fails_soft_with_a_loud_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    second = {**HYPERPOD_WORKSPACE, "id": "g-second0002", "name": "team-grafana"}
    runner = Runner([HYPERPOD_WORKSPACE, second])

    result = _ensure(runner, _settings(), tmp_path)

    assert result["status"] == "FAILED"
    assert "g-second0002" in result["reason"]
    err = capsys.readouterr().err
    assert "WARNING" in err and "Grafana" in err and "g-second0002" in err


def test_a_wrong_operator_workspace_id_fails_the_deploy(tmp_path: Path) -> None:
    runner = Runner([HYPERPOD_WORKSPACE])
    settings = _settings(workspace_id="g-typo", workspace_id_is_operator_input=True)

    with pytest.raises(BootstrapError, match="g-typo"):
        _ensure(runner, settings, tmp_path)


def test_a_persisted_workspace_id_that_vanished_fails_soft(tmp_path: Path) -> None:
    runner = Runner([])
    settings = _settings(workspace_id="g-gone", workspace_id_is_operator_input=False)

    result = _ensure(runner, settings, tmp_path)

    assert result["status"] == "FAILED"
    assert "g-gone" in result["reason"]


def test_a_refused_http_call_fails_soft_but_keeps_the_created_workspace(
    tmp_path: Path,
) -> None:
    """A workspace we created must reach the registry even when the import failed,
    or uninstall would leave it behind."""

    runner = Runner([])
    http = Http({("GET", "/api/org"): HttpResponse(401, "Unauthorized")})

    result = _ensure(runner, _settings(), tmp_path, http=http)

    assert result["status"] == "FAILED"
    assert result["workspace_id"] == "g-created01"
    assert result["ownership"] == "CREATED"
    assert "401" in result["reason"]


def test_a_probe_reproves_a_provisioned_workspace_read_only(tmp_path: Path) -> None:
    runner = Runner([_tagged(HYPERPOD_WORKSPACE, **{"gpu-fault:site-id": SITE})])
    previous = {"status": "PROVISIONED", "workspace_id": "g-5b81a13d97"}

    result = _ensure(
        ReadOnlyProbeRunner(runner),
        _settings(previous=previous),
        tmp_path,
        probe_only=True,
    )

    assert result["status"] == "PROBED"
    assert "create-workspace-service-account-token" not in runner.operations()


@pytest.mark.parametrize(
    "previous",
    [None, {"status": "FAILED", "reason": "ambiguous"}],
    ids=["never-provisioned", "failed-last-run"],
)
def test_a_probe_asks_for_ensure_until_the_dashboards_are_provisioned(
    tmp_path: Path, previous: dict[str, Any] | None
) -> None:
    runner = Runner([_tagged(HYPERPOD_WORKSPACE, **{"gpu-fault:site-id": SITE})])

    with pytest.raises(BootstrapMutationRequired):
        _ensure(
            ReadOnlyProbeRunner(runner),
            _settings(previous=previous),
            tmp_path,
            probe_only=True,
        )


def test_a_probe_whose_workspace_vanished_asks_for_ensure_not_failure(
    tmp_path: Path,
) -> None:
    runner = Runner([])
    previous = {"status": "PROVISIONED", "workspace_id": "g-5b81a13d97"}

    with pytest.raises(BootstrapMutationRequired):
        _ensure(
            ReadOnlyProbeRunner(runner),
            _settings(previous=previous, workspace_id="g-5b81a13d97"),
            tmp_path,
            probe_only=True,
        )


# --- settings, CLI plumbing and site persistence ----------------------------------


def _request(**overrides: Any) -> BootstrapRequest:
    values: dict[str, Any] = {
        "cpu_cluster_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
        "gpu_cluster_arns": ("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",),
        "repository_root": Path("/repo"),
        "state_dir": Path("/state"),
        **overrides,
    }
    return BootstrapRequest(**values)


def test_request_defaults_leave_the_workspace_and_the_viewer_unset() -> None:
    request = _request()

    assert request.grafana_workspace_id is None
    assert request.grafana_viewer is None
    assert not hasattr(request, "grafana_enabled"), "the --grafana mode is gone"
    assert not hasattr(request, "grafana_create"), "the --grafana mode is gone"


def test_settings_take_the_operator_id_over_the_persisted_site_id(
    tmp_path: Path,
) -> None:
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id=SITE)
    state.record(
        "monitoring_install",
        {"grafana": {"status": "PROVISIONED", "workspace_id": "g-5b81a13d97"}},
    )
    existing_site = {"spec": {"health": {"grafanaWorkspaceId": "g-persisted"}}}

    from_option = grafana_settings(
        _request(grafana_workspace_id="g-option"),
        existing_site=existing_site,
        state=state,
    )
    from_site = grafana_settings(_request(), existing_site=existing_site, state=state)
    fresh = grafana_settings(_request(), existing_site=None, state=state)

    assert (from_option.workspace_id, from_option.workspace_id_is_operator_input) == (
        "g-option",
        True,
    )
    assert (from_site.workspace_id, from_site.workspace_id_is_operator_input) == (
        "g-persisted",
        False,
    )
    assert fresh.workspace_id is None
    assert fresh.previous == {"status": "PROVISIONED", "workspace_id": "g-5b81a13d97"}
    assert (
        grafana_settings(
            _request(grafana_viewer="u-42"), existing_site=None, state=state
        ).viewer_sso_user_id
        == "u-42"
    )


def test_cli_options_are_parsed_and_carried_to_the_next_hop_as_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = argparse.ArgumentParser()
    admin_grafana.add_grafana_arguments(parser)
    monkeypatch.delenv(GRAFANA_WORKSPACE_ID_ENV, raising=False)
    monkeypatch.delenv(GRAFANA_VIEWER_ENV, raising=False)

    default = parser.parse_args([])
    explicit = parser.parse_args(
        ["--grafana-workspace-id", "g-5b81a13d97", "--grafana-viewer", "u-42"]
    )

    assert grafana_environment(default) == {}, "no option, no variable"
    assert grafana_request_fields(default) == {
        "grafana_workspace_id": None,
        "grafana_viewer": None,
    }
    assert grafana_environment(explicit) == {
        GRAFANA_WORKSPACE_ID_ENV: "g-5b81a13d97",
        GRAFANA_VIEWER_ENV: "u-42",
    }
    assert grafana_request_fields(explicit) == {
        "grafana_workspace_id": "g-5b81a13d97",
        "grafana_viewer": "u-42",
    }
    for retired in (["--grafana", "disabled"], ["--grafana", "create"]):
        with pytest.raises(SystemExit):
            parser.parse_args(retired)


def test_the_inner_hop_reads_the_variables_the_public_command_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = argparse.ArgumentParser()
    admin_grafana.add_grafana_arguments(parser)
    monkeypatch.setenv(GRAFANA_WORKSPACE_ID_ENV, "g-5b81a13d97")
    monkeypatch.setenv(GRAFANA_VIEWER_ENV, "u-42")

    fields = grafana_request_fields(parser.parse_args([]))
    option_wins = grafana_request_fields(
        parser.parse_args(["--grafana-workspace-id", "g-other0001"])
    )

    assert fields == {"grafana_workspace_id": "g-5b81a13d97", "grafana_viewer": "u-42"}
    assert option_wins["grafana_workspace_id"] == "g-other0001"
    assert option_wins["grafana_viewer"] == "u-42"


def test_site_health_document_persists_the_resolved_workspace(tmp_path: Path) -> None:
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id=SITE)
    state.record(
        "monitoring_install",
        {"grafana": {"status": "PROVISIONED", "workspace_id": "g-5b81a13d97"}},
    )

    provisioned = admin_grafana.grafana_site_health(state, _settings())
    state.record("monitoring_install", {"grafana": {"status": "FAILED", "reason": "x"}})
    failed = admin_grafana.grafana_site_health(
        state, _settings(workspace_id="g-persisted")
    )
    unresolved = admin_grafana.grafana_site_health(state, _settings())

    assert provisioned == {"grafanaWorkspaceId": "g-5b81a13d97"}
    assert failed == {"grafanaWorkspaceId": "g-persisted"}
    assert unresolved == {}, "no workspace yet, nothing to persist"


# --- registry records and checkpoint assets ----------------------------------------


def _records(grafana: dict[str, Any]) -> dict[str, Any]:
    resources = grafana_installation_resources(
        site_id=SITE,
        region=REGION,
        account_id="123456789012",
        state={"monitoring_install": {"grafana": grafana}},
    )
    return {resource.resource_key: resource for resource in resources}


def test_the_hyperpod_owned_workspace_is_registered_external_and_preserved() -> None:
    by_key = _records(
        {
            "status": "PROVISIONED",
            "workspace_id": "g-5b81a13d97",
            "ownership": "EXTERNAL",
            "endpoint": HYPERPOD_WORKSPACE["endpoint"],
            "service_account_id": "9",
            "dashboards": [{"uid": "a", "title": "A", "version": 1}],
        }
    )

    workspace = by_key["aws/grafana/workspace"]
    assert workspace.resource_type == "grafana_workspace"
    assert workspace.resource_id == "g-5b81a13d97"
    assert workspace.ownership is InstallationResourceOwnership.EXTERNAL
    assert workspace.delete_policy is InstallationResourceDeletePolicy.PRESERVE
    assert workspace.attributes["endpoint"] == HYPERPOD_WORKSPACE["endpoint"]
    account = by_key["aws/grafana/service-account"]
    assert account.resource_type == "grafana_service_account"
    assert account.resource_id == "9"
    assert account.attributes["workspace_id"] == "g-5b81a13d97"
    assert account.ownership is InstallationResourceOwnership.CREATED
    assert account.delete_policy is InstallationResourceDeletePolicy.DELETE
    assert account.dependencies == ["aws/grafana/workspace"], (
        "our service account must go before the workspace it lives in"
    )
    assert "aws/grafana/workspace-role" not in by_key


def test_an_adopted_workspace_is_never_promoted_to_created() -> None:
    """``REUSED`` is adopted for deletion by uninstall; a tagged workspace we did
    not create must therefore reach the registry as ``EXTERNAL``."""

    by_key = _records(
        {"status": "PROVISIONED", "workspace_id": "g-5b81a13d97", "ownership": "REUSED"}
    )

    assert by_key["aws/grafana/workspace"].ownership is (
        InstallationResourceOwnership.EXTERNAL
    )
    assert by_key["aws/grafana/workspace"].delete_policy is (
        InstallationResourceDeletePolicy.PRESERVE
    )


def test_a_created_workspace_and_its_role_are_deleted_role_last() -> None:
    by_key = _records(
        {
            "status": "FAILED",
            "reason": "401",
            "workspace_id": "g-created01",
            "ownership": "CREATED",
            "role_name": "gpu-fault-site-a-grafana",
            "role_arn": "arn:aws:iam::123456789012:role/gpu-fault-site-a-grafana",
            "role_ownership": "CREATED",
        }
    )

    workspace = by_key["aws/grafana/workspace"]
    assert workspace.ownership is InstallationResourceOwnership.CREATED
    assert workspace.delete_policy is InstallationResourceDeletePolicy.DELETE
    assert workspace.dependencies == ["aws/grafana/workspace-role"]
    role = by_key["aws/grafana/workspace-role"]
    assert role.resource_type == "iam_role"
    assert role.resource_id == "gpu-fault-site-a-grafana"
    assert role.delete_policy is InstallationResourceDeletePolicy.DELETE


@pytest.mark.parametrize(
    "grafana",
    [{"status": "SKIPPED"}, {"status": "FAILED", "reason": "ambiguous"}],
    ids=["skipped", "failed-before-resolution"],
)
def test_no_workspace_means_no_record(grafana: dict[str, Any]) -> None:
    assert _records(grafana) == {}
    assert (
        grafana_installation_resources(
            site_id=SITE, region=REGION, account_id="123456789012", state={}
        )
        == []
    )


def _datasource_uids(node: Any, found: set[str]) -> set[str]:
    if isinstance(node, dict):
        datasource = node.get("datasource")
        if isinstance(datasource, dict) and "uid" in datasource:
            found.add(str(datasource["uid"]))
        for value in node.values():
            _datasource_uids(value, found)
    elif isinstance(node, list):
        for value in node:
            _datasource_uids(value, found)
    return found


def test_the_shipped_dashboards_honour_the_provisioning_contract() -> None:
    """Every dashboard the repository ships must import into the folder this
    module creates and read through the data source it creates."""

    repository_root = Path(__file__).resolve().parents[2]
    directory = repository_root / admin_grafana.DASHBOARDS_DIRECTORY
    if not directory.is_dir():
        pytest.skip("no dashboards shipped yet")

    models = admin_grafana.load_dashboards(directory)

    assert models, "the dashboards directory exists but holds no importable model"
    assert [model["uid"] for model in models] == sorted(
        path.stem for path in directory.glob("*.json")
    ), "a dashboard file is not named after its uid"
    for model in models:
        assert model["id"] is None, f"{model['uid']} carries an instance-bound id"
        assert _datasource_uids(model, set()) == {DATASOURCE_UID}, (
            f"{model['uid']} reads a data source other than {DATASOURCE_UID}"
        )


def test_dashboard_assets_are_digested_per_file(tmp_path: Path) -> None:
    assert dashboard_asset_digests(tmp_path) == {}
    directory = _dashboards(tmp_path, "gpu-fault-overview")
    (directory / "notes.txt").write_text("ignored", encoding="utf-8")

    first = dashboard_asset_digests(tmp_path)
    (directory / "gpu-fault-overview.json").write_text("{}", encoding="utf-8")
    second = dashboard_asset_digests(tmp_path)

    assert list(first) == ["gpu-fault-overview.json"]
    assert first["gpu-fault-overview.json"] != second["gpu-fault-overview.json"]
