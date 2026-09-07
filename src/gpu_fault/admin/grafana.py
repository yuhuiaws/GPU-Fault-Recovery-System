"""Amazon Managed Grafana dashboards for the administrator deploy path.

The alerting path (``/metrics`` -> ADOT -> AMP -> Alertmanager -> SNS) is
installed by ``install_monitoring``; this module adds the visualisation on top of
it. It resolves one Amazon Managed Grafana workspace in the CPU cluster's region,
creates a Prometheus data source that reads the site's AMP workspace through the
workspace IAM role (SigV4, ``default`` auth type), a folder, and imports every
``deploy/observability/dashboards/*.json`` with ``overwrite`` so a re-run
converges instead of duplicating.

Two rules shape the failure handling. The workspace is never guessed between
candidates: an untagged region with one ACTIVE workspace is adopted and tagged
for the next run, two untagged workspaces are an error that names them, none is
an error that names the flag. And dashboards are not the alerting path: unless
the operator's own input was wrong (``--grafana-workspace-id`` pointing at
nothing, or ``--grafana create`` refused), a failed step is recorded as ``FAILED``
with a loud warning and the deploy continues.

The Grafana HTTP API is reached with a service-account token minted through the
AWS API for one run (15 minutes) and deleted in a ``finally``; the token never
enters state, the summary or the command log (the runner treats the minting
command as sensitive).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    BootstrapMutationRequired,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    safe_name,
    tag_map,
)
from gpu_fault.admin.resource_records import (
    foundation_ownership,
    policy,
    record,
)
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)

DASHBOARDS_DIRECTORY = "deploy/observability/dashboards"
DATASOURCE_UID = "gpu-fault-amp"
DASHBOARD_FOLDER_UID = "gpu-fault-recovery"
DASHBOARD_FOLDER_TITLE = "GPU Fault Recovery"
SERVICE_ACCOUNT_NAME = "gpu-fault-provisioner"
TOKEN_SECONDS_TO_LIVE = 900
# Set on the workspaces this tool created, next to the site tag: a workspace that
# only carries the site tag was adopted, and adopted workspaces are preserved.
CREATED_TAG_KEY = "gpu-fault:grafana-created-by"
CREATED_TAG_VALUE = "gpu-fault-admin"
DRY_RUN_WORKSPACE_ID = "g-dryrun"
GRAFANA_MODES = ("enabled", "disabled", "create")
GRAFANA_ENV = "GPU_FAULT_ADMIN_GRAFANA"
GRAFANA_WORKSPACE_ID_ENV = "GPU_FAULT_ADMIN_GRAFANA_WORKSPACE_ID"
HTTP_TIMEOUT_SECONDS = 30
WORKSPACE_ACTIVE_TIMEOUT_SECONDS = 600
_WORKSPACE_TASK = "monitoring_install"


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: str


# (method, url, headers, body) -> response; the default is urllib, tests inject
# a recorder. Non-2xx statuses are returned, not raised, so the caller can treat
# a 404 as "absent" and everything else as the specific error it is.
HttpTransport = Callable[[str, str, Mapping[str, str], bytes | None], HttpResponse]


@dataclass(frozen=True)
class GrafanaSettings:
    """What one deploy run wants from Grafana, resolved from CLI, site and state."""

    enabled: bool = True
    workspace_id: str | None = None
    # ``--grafana-workspace-id`` on this command is operator input: a wrong value
    # fails the deploy. The same id read back from ``site.yaml`` is not.
    workspace_id_is_operator_input: bool = False
    create: bool = False
    # The previous run's ``monitoring_install.grafana`` record, so the read-only
    # probe can ask for ensure until the dashboards have actually landed.
    previous: Mapping[str, Any] | None = None


# --- CLI plumbing ----------------------------------------------------------------


def add_grafana_arguments(deploy: argparse.ArgumentParser) -> None:
    deploy.add_argument(
        "--grafana",
        choices=GRAFANA_MODES,
        default=None,
        help=(
            "import the solution dashboards into Amazon Managed Grafana in the "
            "CPU cluster's region (default: enabled); 'create' also creates a "
            "workspace when the region has none"
        ),
    )
    deploy.add_argument(
        "--grafana-workspace-id",
        metavar="WORKSPACE_ID",
        help=(
            "existing Amazon Managed Grafana workspace to import into "
            "(default: the one ACTIVE workspace in the region)"
        ),
    )


def grafana_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """The options as variables for the later hops of a public deploy.

    The public command hands off to the source preparer, which re-invokes the
    CLI inside a prepared snapshot without forwarding new arguments; every hop
    inherits its environment, so the options travel the way the schema-change
    consent does.
    """

    environment: dict[str, str] = {}
    mode = getattr(arguments, "grafana", None)
    if mode:
        environment[GRAFANA_ENV] = str(mode)
    workspace_id = getattr(arguments, "grafana_workspace_id", None)
    if workspace_id:
        environment[GRAFANA_WORKSPACE_ID_ENV] = str(workspace_id)
    return environment


def grafana_request_fields(arguments: argparse.Namespace) -> dict[str, Any]:
    """``BootstrapRequest`` fields: the option wins, then the inherited variable."""

    mode = str(
        getattr(arguments, "grafana", None)
        or os.getenv("GPU_FAULT_ADMIN_GRAFANA", "enabled")
    ).strip()
    if mode not in GRAFANA_MODES:
        raise BootstrapError(
            f"GPU_FAULT_ADMIN_GRAFANA must be one of {', '.join(GRAFANA_MODES)}, "
            f"not {mode!r}"
        )
    workspace_id = str(
        getattr(arguments, "grafana_workspace_id", None)
        or os.getenv("GPU_FAULT_ADMIN_GRAFANA_WORKSPACE_ID", "")
    ).strip()
    return {
        "grafana_enabled": mode != "disabled",
        "grafana_workspace_id": workspace_id or None,
        "grafana_create": mode == "create",
    }


def grafana_settings(
    request: BootstrapRequest,
    *,
    existing_site: Mapping[str, Any] | None,
    state: BootstrapState,
) -> GrafanaSettings:
    previous = _state_result(state)
    if request.grafana_workspace_id:
        return GrafanaSettings(
            enabled=request.grafana_enabled,
            workspace_id=request.grafana_workspace_id,
            workspace_id_is_operator_input=True,
            create=request.grafana_create,
            previous=previous,
        )
    persisted = _site_health(existing_site).get("grafanaWorkspaceId")
    return GrafanaSettings(
        enabled=request.grafana_enabled,
        workspace_id=str(persisted) if persisted else None,
        create=request.grafana_create,
        previous=previous,
    )


def grafana_site_health(
    state: BootstrapState, settings: GrafanaSettings
) -> dict[str, Any]:
    """The ``spec.health`` keys that persist this run's Grafana decision."""

    if not settings.enabled:
        return {"grafanaEnabled": False}
    result = _state_result(state) or {}
    workspace_id = result.get("workspace_id") or settings.workspace_id
    health: dict[str, Any] = {"grafanaEnabled": True}
    if workspace_id:
        health["grafanaWorkspaceId"] = str(workspace_id)
    return health


def _state_result(state: BootstrapState) -> Mapping[str, Any] | None:
    resources = state.value.get("resources")
    task = resources.get(_WORKSPACE_TASK) if isinstance(resources, Mapping) else None
    grafana = task.get("grafana") if isinstance(task, Mapping) else None
    return cast(Mapping[str, Any], grafana) if isinstance(grafana, Mapping) else None


def _site_health(site: Mapping[str, Any] | None) -> Mapping[str, Any]:
    spec = site.get("spec") if isinstance(site, Mapping) else None
    health = spec.get("health") if isinstance(spec, Mapping) else None
    return cast(Mapping[str, Any], health) if isinstance(health, Mapping) else {}


# --- workspace resolution ----------------------------------------------------------


def ensure_grafana_workspace(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
    requested_id: str | None = None,
    create_allowed: bool = False,
) -> dict[str, Any]:
    """Resolve the one workspace this site uses; see the module docstring."""

    region = cpu.region
    if requested_id:
        workspace = _describe_workspace(runner, region, requested_id)
        if workspace is None:
            raise BootstrapError(
                f"Amazon Managed Grafana workspace {requested_id} does not exist "
                f"in {region}"
            )
        _require_active(workspace)
        return _resolved(workspace, region, _ownership_of(workspace, site_id))
    workspaces = cast(
        list[dict[str, Any]],
        runner.aws_json(region, "grafana", "list-workspaces").get("workspaces", []),
    )
    tagged = [
        item for item in workspaces if tag_map(item.get("tags")).get(SITE_TAG_KEY)
    ]
    ours = [item for item in tagged if _site_of(item) == site_id]
    if ours:
        workspace = _describe_workspace(runner, region, str(ours[0]["id"])) or ours[0]
        _require_active(workspace)
        return _resolved(workspace, region, _ownership_of(workspace, site_id))
    candidates = [
        item
        for item in workspaces
        if item not in tagged and str(item.get("status") or "") == "ACTIVE"
    ]
    if len(candidates) > 1:
        listed = ", ".join(
            f"{item['id']} ({item.get('name') or 'unnamed'})" for item in candidates
        )
        raise BootstrapError(
            f"{len(candidates)} Amazon Managed Grafana workspaces in {region} carry "
            f"no {SITE_TAG_KEY} tag: {listed}; pass --grafana-workspace-id to "
            "choose one, or --grafana disabled"
        )
    if candidates:
        workspace = candidates[0]
        runner.run(
            [
                "aws",
                "grafana",
                "tag-resource",
                "--region",
                region,
                "--resource-arn",
                _workspace_arn(cpu, str(workspace["id"])),
                "--tags",
                f"{SITE_TAG_KEY}={site_id}",
            ],
            mutate=True,
            capture=False,
        )
        return _resolved(workspace, region, "EXTERNAL")
    if runner.dry_run:
        return _resolved(
            {"id": DRY_RUN_WORKSPACE_ID, "status": "ACTIVE"}, region, "EXTERNAL"
        )
    if not create_allowed:
        raise BootstrapError(
            f"no Amazon Managed Grafana workspace in {region}; "
            "pass --grafana-workspace-id, or --grafana disabled"
        )
    return _create_workspace(runner, cpu=cpu, site_id=site_id)


def _describe_workspace(
    runner: CommandRunner, region: str, workspace_id: str
) -> dict[str, Any] | None:
    try:
        document = runner.aws_json(
            region, "grafana", "describe-workspace", "--workspace-id", workspace_id
        )
    except BootstrapError as exc:
        if "ResourceNotFoundException" in str(exc):
            return None
        raise
    return cast(dict[str, Any], document.get("workspace") or {})


def _require_active(workspace: Mapping[str, Any]) -> None:
    status = str(workspace.get("status") or "UNKNOWN")
    if status != "ACTIVE":
        raise BootstrapError(
            f"Amazon Managed Grafana workspace {workspace.get('id')} is {status}, "
            "not ACTIVE"
        )


def _site_of(workspace: Mapping[str, Any]) -> str | None:
    return tag_map(workspace.get("tags")).get(SITE_TAG_KEY)


def _ownership_of(workspace: Mapping[str, Any], site_id: str) -> str:
    tags = tag_map(workspace.get("tags"))
    owner = tags.get(SITE_TAG_KEY)
    if owner is None:
        return "EXTERNAL"
    if owner != site_id:
        raise BootstrapError(
            f"Amazon Managed Grafana workspace {workspace.get('id')} belongs to "
            f"site {owner!r}, not {site_id!r}; refusing to share it"
        )
    if tags.get(CREATED_TAG_KEY) == CREATED_TAG_VALUE:
        return "CREATED"
    return "REUSED"


def _workspace_arn(cpu: ClusterIdentity, workspace_id: str) -> str:
    return f"arn:aws:grafana:{cpu.region}:{cpu.account_id}:/workspaces/{workspace_id}"


def _resolved(
    workspace: Mapping[str, Any], region: str, ownership: str
) -> dict[str, Any]:
    return {
        "workspace_id": str(workspace["id"]),
        "name": str(workspace.get("name") or ""),
        "endpoint": str(workspace.get("endpoint") or ""),
        "role_arn": workspace.get("workspaceRoleArn"),
        "ownership": ownership,
        "region": region,
    }


def _create_workspace(
    runner: CommandRunner, *, cpu: ClusterIdentity, site_id: str
) -> dict[str, Any]:
    role = _ensure_workspace_role(runner, cpu=cpu, site_id=site_id)
    tags = {SITE_TAG_KEY: site_id, CREATED_TAG_KEY: CREATED_TAG_VALUE}
    created = runner.aws_json(
        cpu.region,
        "grafana",
        "create-workspace",
        "--workspace-name",
        safe_name(f"gpu-fault-{site_id}", maximum=255),
        "--account-access-type",
        "CURRENT_ACCOUNT",
        "--authentication-providers",
        "AWS_SSO",
        "--permission-type",
        "CUSTOMER_MANAGED",
        "--workspace-role-arn",
        role["role_arn"],
        "--workspace-data-sources",
        "PROMETHEUS",
        "--tags",
        json.dumps(tags, separators=(",", ":")),
        mutate=True,
    )
    workspace_id = str((created.get("workspace") or {}).get("id") or "")
    if not workspace_id:
        raise BootstrapError("create-workspace returned no workspace id")
    deadline = time.monotonic() + WORKSPACE_ACTIVE_TIMEOUT_SECONDS
    while True:
        workspace = _describe_workspace(runner, cpu.region, workspace_id) or {}
        status = str(workspace.get("status") or "")
        if status == "ACTIVE":
            break
        if status in {"CREATION_FAILED", "FAILED", "DELETING", "DELETED"}:
            raise BootstrapError(
                f"Amazon Managed Grafana workspace {workspace_id} entered {status}"
            )
        if time.monotonic() >= deadline:
            raise BootstrapError(
                f"Amazon Managed Grafana workspace {workspace_id} did not become ACTIVE"
            )
        time.sleep(5)
    return {**_resolved(workspace, cpu.region, "CREATED"), **role}


def _ensure_workspace_role(
    runner: CommandRunner, *, cpu: ClusterIdentity, site_id: str
) -> dict[str, Any]:
    """The IAM role a created workspace queries AMP with (read-only)."""

    role_name = safe_name(f"gpu-fault-{site_id}-grafana", maximum=64)
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "grafana.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": cpu.account_id},
                    "StringLike": {"aws:SourceArn": _workspace_arn(cpu, "*")},
                },
            }
        ],
    }
    document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aps:QueryMetrics",
                    "aps:GetSeries",
                    "aps:GetLabels",
                    "aps:GetMetricMetadata",
                    "aps:ListRules",
                    "aps:ListAlertManagerAlerts",
                    "aps:ListAlertManagerAlertGroups",
                    "aps:ListAlertManagerSilences",
                    "aps:GetAlertManagerStatus",
                    "aps:DescribeWorkspace",
                ],
                "Resource": f"arn:aws:aps:{cpu.region}:{cpu.account_id}:workspace/*",
            },
            {"Effect": "Allow", "Action": ["aps:ListWorkspaces"], "Resource": "*"},
        ],
    }
    try:
        existing = runner.aws_json(
            cpu.region, "iam", "get-role", "--role-name", role_name
        )
    except BootstrapError as exc:
        if "NoSuchEntity" not in str(exc):
            raise
        existing = {}
    ownership = "REUSED"
    role_arn = str((existing.get("Role") or {}).get("Arn") or "")
    if not role_arn:
        ownership = "CREATED"
        created = runner.aws_json(
            cpu.region,
            "iam",
            "create-role",
            "--role-name",
            role_name,
            "--assume-role-policy-document",
            json.dumps(trust, separators=(",", ":")),
            "--tags",
            f"Key={SITE_TAG_KEY},Value={site_id}",
            mutate=True,
        )
        role_arn = str(
            (created.get("Role") or {}).get("Arn")
            or f"arn:aws:iam::{cpu.account_id}:role/{role_name}"
        )
    runner.run(
        [
            "aws",
            "iam",
            "put-role-policy",
            "--role-name",
            role_name,
            "--policy-name",
            "gpu-fault-grafana-amp-read",
            "--policy-document",
            json.dumps(document, separators=(",", ":")),
        ],
        mutate=True,
        capture=False,
    )
    return {"role_name": role_name, "role_arn": role_arn, "role_ownership": ownership}


# --- dashboards over the Grafana HTTP API ------------------------------------------


def load_dashboards(directory: Path) -> list[dict[str, Any]]:
    """Every ``*.json`` under ``directory`` as a Grafana dashboard model.

    A dashboard needs a stable ``uid`` (that is what makes ``overwrite`` an
    update rather than a duplicate) and a ``title``; a numeric ``id`` binds the
    model to the Grafana instance it was exported from, so it is dropped.
    """

    if not directory.is_dir():
        return []
    dashboards = []
    for path in sorted(directory.glob("*.json")):
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise BootstrapError(f"dashboard {path.name} is not valid JSON") from exc
        uid = model.get("uid") if isinstance(model, dict) else None
        title = model.get("title") if isinstance(model, dict) else None
        if (
            not isinstance(uid, str)
            or not uid
            or not isinstance(title, str)
            or not title
        ):
            raise BootstrapError(
                f"dashboard {path.name} needs a string uid and title "
                f"(under {DASHBOARDS_DIRECTORY})"
            )
        dashboards.append({**model, "id": None})
    return dashboards


def dashboard_asset_digests(repository_root: Path) -> dict[str, str]:
    """Per-file digests of the dashboard assets, for the task checkpoint."""

    directory = repository_root / DASHBOARDS_DIRECTORY
    if not directory.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.json"))
    }


def urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> HttpResponse:
    request = urllib.request.Request(
        url, data=body, method=method, headers=dict(headers)
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return HttpResponse(
                int(response.status), response.read().decode("utf-8", "replace")
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(int(exc.code), exc.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError) as exc:
        raise BootstrapError(f"Grafana {method} {url} failed: {exc}") from None


class _GrafanaApi:
    def __init__(self, base_url: str, token: str, transport: HttpTransport) -> None:
        self._base_url = base_url
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._transport = transport

    def call(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        accept: tuple[int, ...] = (200,),
    ) -> tuple[int, Any]:
        body = (
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        )
        response = self._transport(method, self._base_url + path, self._headers, body)
        if response.status not in accept:
            raise BootstrapError(
                f"Grafana {method} {path} returned {response.status}: "
                f"{response.body[:300]}"
            )
        try:
            return response.status, json.loads(response.body) if response.body else {}
        except ValueError:
            return response.status, {}


def _datasource_document(amp_workspace_id: str, region: str) -> dict[str, Any]:
    return {
        "uid": DATASOURCE_UID,
        "name": f"GPU Fault AMP ({region})",
        "type": "prometheus",
        "access": "proxy",
        "url": (
            f"https://aps-workspaces.{region}.amazonaws.com/workspaces/"
            f"{amp_workspace_id}"
        ),
        "isDefault": False,
        "jsonData": {
            "httpMethod": "POST",
            "sigV4Auth": True,
            "sigV4AuthType": "default",
            "sigV4Region": region,
            "manageAlerts": True,
            "prometheusType": "Prometheus",
        },
    }


def _upsert_datasource(api: _GrafanaApi, document: Mapping[str, Any]) -> None:
    path = f"/api/datasources/uid/{DATASOURCE_UID}"
    status, _existing = api.call("GET", path, accept=(200, 404))
    if status == 404:
        api.call("POST", "/api/datasources", document)
    else:
        api.call("PUT", path, document)


def _verify_datasource(api: _GrafanaApi, amp_workspace_id: str) -> None:
    """Prove the data source can reach AMP through the workspace role.

    The per-data-source health endpoint is the direct answer; some Grafana
    versions answer it with a 400 for a healthy SigV4 source, so an instant
    ``up`` query is the second opinion before the step is called failed.
    """

    try:
        _status, health = api.call(
            "GET", f"/api/datasources/uid/{DATASOURCE_UID}/health"
        )
        if str((health or {}).get("status") or "OK").upper() == "OK":
            return
        detail = f"health reported {json.dumps(health)[:300]}"
    except BootstrapError as exc:
        detail = str(exc)
    query = {
        "from": "now-5m",
        "to": "now",
        "queries": [
            {
                "refId": "A",
                "datasource": {"uid": DATASOURCE_UID},
                "expr": "up",
                "instant": True,
            }
        ],
    }
    try:
        api.call("POST", "/api/ds/query", query)
    except BootstrapError as exc:
        raise BootstrapError(
            f"Grafana data source {DATASOURCE_UID} cannot query AMP workspace "
            f"{amp_workspace_id}: {detail}; query: {exc}"
        ) from None


def _upsert_folder(api: _GrafanaApi) -> None:
    status, _folder = api.call(
        "GET", f"/api/folders/{DASHBOARD_FOLDER_UID}", accept=(200, 404)
    )
    if status == 404:
        api.call(
            "POST",
            "/api/folders",
            {"uid": DASHBOARD_FOLDER_UID, "title": DASHBOARD_FOLDER_TITLE},
        )


def _import_dashboard(api: _GrafanaApi, model: Mapping[str, Any]) -> dict[str, Any]:
    _status, imported = api.call(
        "POST",
        "/api/dashboards/db",
        {
            "dashboard": model,
            "folderUid": DASHBOARD_FOLDER_UID,
            "overwrite": True,
            "message": "gpu-fault-admin deploy",
        },
    )
    return {
        "uid": str(model["uid"]),
        "title": str(model["title"]),
        "version": (imported or {}).get("version"),
    }


def _ensure_service_account(
    runner: CommandRunner, region: str, workspace_id: str
) -> str:
    accounts = cast(
        list[dict[str, Any]],
        runner.aws_json(
            region,
            "grafana",
            "list-workspace-service-accounts",
            "--workspace-id",
            workspace_id,
        ).get("serviceAccounts", []),
    )
    for account in accounts:
        if str(account.get("name") or "") == SERVICE_ACCOUNT_NAME:
            return str(account["id"])
    created = runner.aws_json(
        region,
        "grafana",
        "create-workspace-service-account",
        "--workspace-id",
        workspace_id,
        "--grafana-role",
        "ADMIN",
        "--name",
        SERVICE_ACCOUNT_NAME,
        mutate=True,
    )
    return str(created["id"])


def provision_grafana(
    runner: CommandRunner,
    *,
    workspace: Mapping[str, Any],
    amp_workspace_id: str,
    region: str,
    dashboards_dir: Path,
    http: HttpTransport | None = None,
) -> dict[str, Any]:
    """Data source, folder and dashboards, through a one-run service token."""

    dashboards = load_dashboards(dashboards_dir)
    workspace_id = str(workspace["workspace_id"])
    base_url = f"https://{workspace.get('endpoint') or ''}"
    summary: dict[str, Any] = {
        "datasource_uid": DATASOURCE_UID,
        "folder_uid": DASHBOARD_FOLDER_UID,
        "workspace_url": base_url,
        "dashboards_url": f"{base_url}/dashboards/f/{DASHBOARD_FOLDER_UID}",
    }
    if runner.dry_run:
        return {
            **summary,
            "status": "DRY_RUN",
            "dashboards": [
                {"uid": item["uid"], "title": item["title"], "version": None}
                for item in dashboards
            ],
        }
    account_id = _ensure_service_account(runner, region, workspace_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    minted = runner.aws_json(
        region,
        "grafana",
        "create-workspace-service-account-token",
        "--workspace-id",
        workspace_id,
        "--service-account-id",
        account_id,
        "--name",
        f"{SERVICE_ACCOUNT_NAME}-{stamp}",
        "--seconds-to-live",
        str(TOKEN_SECONDS_TO_LIVE),
        mutate=True,
        sensitive=True,
    )
    token = cast(dict[str, Any], minted.get("serviceAccountToken") or {})
    if not token.get("key"):
        raise BootstrapError("Grafana service-account token was not returned")
    try:
        api = _GrafanaApi(base_url, str(token["key"]), http or urllib_transport)
        # Not `/api/health`: Amazon Managed Grafana answers that path with
        # `400 Not allowed` for every caller (verified live 2026-09-07), so it
        # cannot prove anything. `/api/org` needs the token and proves both the
        # endpoint and the service account in one round trip.
        api.call("GET", "/api/org")
        _upsert_datasource(api, _datasource_document(amp_workspace_id, region))
        _verify_datasource(api, amp_workspace_id)
        _upsert_folder(api)
        imported = [_import_dashboard(api, model) for model in dashboards]
    finally:
        try:
            runner.run(
                [
                    "aws",
                    "grafana",
                    "delete-workspace-service-account-token",
                    "--region",
                    region,
                    "--workspace-id",
                    workspace_id,
                    "--service-account-id",
                    account_id,
                    "--token-id",
                    str(token.get("id") or ""),
                ],
                mutate=True,
                capture=False,
            )
        except BootstrapError as exc:
            # The token expires on its own within the quarter hour; a failed
            # revocation must not mask the provisioning result.
            print(
                f"WARNING: Grafana service-account token was not revoked: {exc}",
                file=sys.stderr,
            )
    return {
        **summary,
        "status": "PROVISIONED",
        "service_account_id": account_id,
        "dashboards": imported,
    }


# --- the step the monitoring task runs -----------------------------------------------


def ensure_grafana_dashboards(
    runner: CommandRunner,
    *,
    settings: GrafanaSettings | None,
    cpu: ClusterIdentity,
    site_id: str,
    amp_workspace_id: str,
    repository_root: Path,
    probe_only: bool = False,
    http: HttpTransport | None = None,
) -> dict[str, Any]:
    """Resolve the workspace and import the dashboards; see the module docstring.

    ``probe_only`` runs with the read-only runner and never mints a token: it
    re-proves that a provisioned workspace still resolves, and asks for ensure
    (``BootstrapMutationRequired``) whenever the last run did not end in
    ``PROVISIONED`` -- a soft failure is retried on every deploy, not cached.
    """

    if settings is None or not settings.enabled:
        return {"status": "DISABLED"}
    previous = dict(settings.previous or {})
    if probe_only:
        if str(previous.get("status") or "") != "PROVISIONED":
            raise BootstrapMutationRequired("grafana dashboards")
        try:
            ensure_grafana_workspace(
                runner,
                cpu=cpu,
                site_id=site_id,
                requested_id=settings.workspace_id
                or str(previous.get("workspace_id") or "")
                or None,
            )
        except BootstrapError as exc:
            raise BootstrapMutationRequired("grafana workspace") from exc
        return {"status": "PROBED"}
    operator_input = settings.workspace_id_is_operator_input or settings.create
    try:
        workspace = ensure_grafana_workspace(
            runner,
            cpu=cpu,
            site_id=site_id,
            requested_id=settings.workspace_id,
            create_allowed=settings.create,
        )
    except BootstrapError as exc:
        if operator_input:
            raise
        return _failed(exc, {})
    try:
        summary = provision_grafana(
            runner,
            workspace=workspace,
            amp_workspace_id=amp_workspace_id,
            region=cpu.region,
            dashboards_dir=repository_root / DASHBOARDS_DIRECTORY,
            http=http,
        )
    except (BootstrapError, ValueError) as exc:
        return _failed(exc, workspace)
    return {"status": "PROVISIONED", **workspace, **summary}


def _failed(error: Exception, workspace: Mapping[str, Any]) -> dict[str, Any]:
    reason = str(error)
    print(
        "WARNING: Grafana dashboards were not provisioned; the alerting path is "
        f"unaffected and the deploy continues. Reason: {reason}. Import "
        f"{DASHBOARDS_DIRECTORY}/*.json by hand or re-run deploy once fixed.",
        file=sys.stderr,
        flush=True,
    )
    return {"status": "FAILED", "reason": reason, **workspace}


# --- installation registry -----------------------------------------------------------


def grafana_installation_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    state: Mapping[str, Any],
) -> list[InstallationResource]:
    """Registry records for the workspace, our service account and our role.

    ``REUSED`` is deliberately mapped to ``EXTERNAL`` (``foundation_ownership``):
    uninstall adopts ``REUSED`` records for deletion, and a workspace we only
    tagged -- the HyperPod observability workspace, typically -- must never be
    deleted by us. Only a workspace carrying our creation tag is ``CREATED``.
    """

    task = state.get(_WORKSPACE_TASK) or {}
    grafana = task.get("grafana") if isinstance(task, Mapping) else None
    if not isinstance(grafana, Mapping) or not grafana.get("workspace_id"):
        return []
    workspace_id = str(grafana["workspace_id"])
    ownership = foundation_ownership(grafana.get("ownership"))
    role_created = (
        grafana.get("role_name")
        and str(grafana.get("role_ownership") or "") == "CREATED"
        and ownership is InstallationResourceOwnership.CREATED
    )
    resources = [
        record(
            site_id=site_id,
            resource_key="aws/grafana/workspace",
            resource_type="grafana_workspace",
            resource_id=workspace_id,
            resource_arn=f"arn:aws:grafana:{region}:{account_id}:/workspaces/{workspace_id}",
            region=region,
            account_id=account_id,
            ownership=ownership,
            delete_policy=policy(ownership),
            dependencies=["aws/grafana/workspace-role"] if role_created else [],
            attributes={
                "endpoint": grafana.get("endpoint"),
                "status": grafana.get("status"),
                "datasource_uid": grafana.get("datasource_uid"),
                "folder_uid": grafana.get("folder_uid"),
                "dashboards": len(grafana.get("dashboards") or ()),
            },
        )
    ]
    if grafana.get("service_account_id"):
        resources.append(
            record(
                site_id=site_id,
                resource_key="aws/grafana/service-account",
                resource_type="grafana_service_account",
                resource_id=str(grafana["service_account_id"]),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                dependencies=["aws/grafana/workspace"],
                attributes={
                    "workspace_id": workspace_id,
                    "name": SERVICE_ACCOUNT_NAME,
                },
            )
        )
    if role_created:
        resources.append(
            record(
                site_id=site_id,
                resource_key="aws/grafana/workspace-role",
                resource_type="iam_role",
                resource_id=str(grafana["role_name"]),
                resource_arn=str(grafana.get("role_arn") or "") or None,
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
            )
        )
    return resources
