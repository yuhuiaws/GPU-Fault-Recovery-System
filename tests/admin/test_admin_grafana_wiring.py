"""How the Grafana step is wired into deploy: CLI, bootstrap, checkpoint, site,
registry.

``test_admin_grafana.py`` proves the step itself; this module proves the deploy
path actually reaches it with the operator's options, re-runs it when the
dashboards change, persists the decision in ``site.yaml`` and registers the
workspace for uninstall.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_services as admin_bootstrap_services
from gpu_fault.admin import bootstrap_tasks
from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import notification_bootstrap as admin_notification_bootstrap
from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import (
    BootstrapRequest,
    BootstrapResult,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.grafana import (
    GRAFANA_VIEWER_ENV,
    GRAFANA_WORKSPACE_ID_ENV,
    GrafanaSettings,
)
from gpu_fault.admin.resource_registry import build_installation_snapshot
from gpu_fault.admin.site import SiteConfigError, load_site
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)
from tests.admin._bootstrap_support import _cluster
from tests.admin.test_admin_site import site_file

SITE = "site-a"


def _gpu(name: str) -> ClusterIdentity:
    return replace(
        _cluster(), role="gpu", hyperpod_name=name, eks_name=name, context=name
    )


# --- bootstrap task plumbing ---------------------------------------------------------


def test_platform_tasks_hand_the_grafana_settings_to_install_monitoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[dict[str, Any]] = []
    settings = GrafanaSettings(workspace_id="g-5b81a13d97")

    def install_monitoring(_runner: Any, **keywords: Any) -> dict[str, Any]:
        received.append(keywords)
        return {}

    monkeypatch.setattr(
        admin_bootstrap_services, "install_monitoring", install_monitoring
    )
    monkeypatch.setattr(
        admin_bootstrap_services, "install_aurora_refresh", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        admin_bootstrap_services, "provision_node_action_keys", lambda *_a, **_k: {}
    )

    # The platform graph reads its foundation inputs from the state at task
    # start, so the two it depends on are recorded as an earlier run would have.
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id=SITE)
    state.record(
        "monitoring_resources",
        {"workspace_id": "ws-a", "sns_topic_arn": "arn:aws:sns:x:1:t"},
    )
    state.complete("monitoring_resources")
    state.record("aurora", {})
    state.complete("aurora")
    bootstrap_tasks.platform_task_graph(
        runner=CommandRunner(),
        state=state,
        repository_root=tmp_path,
        cpu=_cluster(),
        gpu_clusters=[_gpu("gpu-a")],
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id=SITE,
        adot_image="adot@sha256:bbb",
        alert_email="ops@example.com",
        release_manifest=tmp_path / "release.json",
        runtime_image="runtime@sha256:aaa",
        fleet_master_file=tmp_path / "fleet-master",
        ensure_aurora_ready=lambda *_a, **_k: {},
        grafana=settings,
    ).run(state=state)

    assert received and received[0]["grafana"] == settings


def test_install_monitoring_records_the_grafana_step_with_the_amp_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step runs after the installer converged, inside the same task, so its
    result is checkpointed with ``monitoring_install`` and re-proved with it."""

    cpu = _cluster()
    endpoint = (
        "https://aps-workspaces.us-east-1.amazonaws.com/"
        "workspaces/ws-a/api/v1/remote_write"
    )
    received: list[dict[str, Any]] = []

    def ensure_grafana_dashboards(runner: Any, **keywords: Any) -> dict[str, Any]:
        received.append(
            {"read_only": isinstance(runner, ReadOnlyProbeRunner), **keywords}
        )
        return {"status": "PROBED"}

    monkeypatch.setattr(
        admin_bootstrap_services, "ensure_grafana_dashboards", ensure_grafana_dashboards
    )

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            if kwargs.get("mutate"):
                raise AssertionError("probe mutated live state")
            if "jsonpath={.spec.template.spec.containers[0].image}" in arguments:
                return "adot@sha256:bbb"
            if "jsonpath={.status.availableReplicas}" in arguments:
                return "1"
            if "jsonpath={.data}" in arguments:
                return json.dumps({"collector.yaml": f"endpoint: {endpoint}"})
            if "list-rule-groups-namespaces" in arguments:
                return json.dumps({"ruleGroupsNamespaces": [{"name": "capacity"}]})
            if "describe-alert-manager-definition" in arguments:
                return json.dumps({"alertManagerDefinition": {"status": {}}})
            if "list-pod-identity-associations" in arguments:
                return json.dumps({"associations": [{"associationId": "assoc-a"}]})
            if "describe-pod-identity-association" in arguments:
                return json.dumps(
                    {
                        "association": {
                            "associationId": "assoc-a",
                            "roleArn": (
                                "arn:aws:iam::123456789012:role/"
                                "gpu-fault-site-a-amp-writer"
                            ),
                        }
                    }
                )
            if "get-role-policy" in arguments:
                return json.dumps(
                    {
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": ["aps:RemoteWrite"],
                                    "Resource": (
                                        "arn:aws:aps:us-east-1:123456789012:"
                                        "workspace/ws-a"
                                    ),
                                }
                            ],
                        }
                    }
                )
            if "get-role" in arguments:
                return json.dumps(
                    {
                        "Role": {
                            "AssumeRolePolicyDocument": (
                                admin_bootstrap_services.pod_identity_trust(_cluster())
                            ),
                            "Tags": [
                                {
                                    "Key": admin_bootstrap_services.SITE_TAG_KEY,
                                    "Value": SITE,
                                }
                            ],
                        }
                    }
                )
            raise AssertionError(arguments)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=""),
    )
    settings = GrafanaSettings(previous={"status": "PROVISIONED"})

    result = admin_bootstrap_services.install_monitoring(
        ReadOnlyProbeRunner(Runner()),
        repository_root=tmp_path,
        cpu=cpu,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id=SITE,
        monitoring={
            "workspace_id": "ws-a",
            "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:gpu-fault-site-a",
        },
        adot_image="adot@sha256:bbb",
        alert_email="ops@example.com",
        probe_only=True,
        grafana=settings,
    )

    assert result["grafana"] == {"status": "PROBED"}
    (call,) = received
    assert call["read_only"] is True, "the probe reached Grafana with a writing runner"
    assert call["probe_only"] is True
    assert call["settings"] == settings
    assert call["amp_workspace_id"] == "ws-a"
    assert call["site_id"] == SITE
    assert call["repository_root"] == tmp_path
    assert call["cpu"] == cpu


def test_bootstrap_resolves_grafana_from_the_existing_site_and_persists_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cpu = _cluster()
    gpu_a = _gpu("gpu-a")
    existing_site = {"spec": {"health": {"grafanaWorkspaceId": "g-persisted"}}}
    platform: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []

    def record_platform(**keywords: Any) -> None:
        platform.append(keywords)
        keywords["state"].record(
            "monitoring_install",
            {
                "grafana": {
                    "status": "PROVISIONED",
                    "workspace_id": "g-persisted",
                    "region": "us-east-1",
                    "dashboards_url": "https://g-persisted.grafana-workspace/d",
                }
            },
        )

    def record_document(**keywords: Any) -> dict[str, Any]:
        documents.append(keywords)
        return {}

    monkeypatch.setattr(
        admin_bootstrap, "validate_bootstrap_dependencies", lambda: None
    )
    monkeypatch.setattr(
        admin_bootstrap,
        "discover_bootstrap_scope",
        lambda **_kwargs: (existing_site, cpu, [gpu_a]),
    )
    monkeypatch.setattr(admin_bootstrap, "_require_same_scope", lambda *_a, **_k: None)
    monkeypatch.setattr(
        admin_bootstrap, "bootstrap_gpu_scope", lambda *_arguments: [gpu_a]
    )
    monkeypatch.setattr(
        admin_bootstrap,
        "prepare_signed_release",
        lambda *_a, **_k: {
            "manifest": str(tmp_path / "release.json"),
            "images": {"adot": "adot:1", "runtime": "runtime:1"},
        },
    )
    monkeypatch.setattr(
        admin_notification_bootstrap,
        "notification_routing",
        lambda *_a, **_k: ("admin@example.com", {}),
    )
    monkeypatch.setattr(admin_bootstrap, "_update_kubeconfig", lambda *_a, **_k: None)
    monkeypatch.setattr(admin_bootstrap, "_ensure_namespace", lambda *_a, **_k: None)
    monkeypatch.setattr(
        admin_bootstrap, "_initial_secure_files", lambda **_k: ({}, tmp_path / "secure")
    )
    monkeypatch.setattr(
        admin_bootstrap, "_ensure_base_secrets", lambda *_a, **_k: tmp_path / "master"
    )
    monkeypatch.setattr(
        admin_bootstrap_services, "_ensure_pod_identity_agent", lambda *_a: {}
    )
    monkeypatch.setattr(
        admin_bootstrap, "revalidate_pod_identity_agent", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        admin_bootstrap, "bootstrap_aurora_capacity", lambda _state_dir: None
    )
    monkeypatch.setattr(admin_bootstrap, "foundation_task_graph", lambda **_k: None)
    # The platform graph builder is what receives the resolved Grafana settings.
    monkeypatch.setattr(admin_bootstrap, "platform_task_graph", record_platform)
    monkeypatch.setattr(
        admin_bootstrap,
        "run_bootstrap_tasks",
        lambda **_k: {
            "executor_role:gpu-a": {"role_arn": "arn:aws:iam::1:role/gpu-a"},
            "aurora": {},
            "monitoring_resources": {},
            "nlb_network": {},
            "pki": {},
        },
    )
    monkeypatch.setattr(admin_bootstrap, "_site_document", record_document)
    monkeypatch.setattr(
        admin_bootstrap,
        "finalize_bootstrap_site",
        lambda *_a, **_k: BootstrapResult(
            site_file=tmp_path / "site.yaml", state_file=tmp_path / "state.json"
        ),
    )

    admin_bootstrap.bootstrap_from_arns(
        BootstrapRequest(
            cpu_cluster_arn=cpu.input_arn,
            gpu_cluster_arns=(gpu_a.input_arn,),
            repository_root=tmp_path / "repo",
            state_dir=tmp_path / "state",
        )
    )

    (platform_call,) = platform
    settings = platform_call["grafana"]
    assert isinstance(settings, GrafanaSettings), (
        "the platform tasks did not receive resolved Grafana settings"
    )
    assert settings.workspace_id == "g-persisted"
    assert settings.workspace_id_is_operator_input is False, (
        "an id read back from site.yaml was treated as this command's input"
    )
    (document_call,) = documents
    assert document_call["grafana_health"] == {"grafanaWorkspaceId": "g-persisted"}
    err = capsys.readouterr().err
    assert "Grafana dashboards:" in err, (
        "the deploy did not say where the dashboards are"
    )
    assert "aws grafana update-permissions" in err and "g-persisted" in err, (
        "the deploy did not print the exact permission command for the workspace"
    )


# --- checkpoint ------------------------------------------------------------------


def _bind(
    tmp_path: Path, root: Path, *, request_overrides: dict[str, Any] | None = None
) -> dict[str, str]:
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "bootstrap-state.json").unlink(missing_ok=True)
    manifest = state_dir / "release.json"
    manifest.write_text(json.dumps({"release_id": "0100-a"}), encoding="utf-8")
    state = BootstrapState(state_dir / "bootstrap-state.json", site_id=SITE)
    request = BootstrapRequest(
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        gpu_cluster_arns=("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",),
        repository_root=root,
        state_dir=state_dir,
        alert_email="ops@example.com",
        **(request_overrides or {}),
    )
    bind_bootstrap_inputs(
        state,
        request=request,
        cpu=_cluster(),
        gpu_clusters=(_gpu("gpu-a"),),
        release={
            "release_id": "0100-a",
            "manifest": str(manifest),
            "agent_config_digest": "c" * 64,
            "images": {"runtime": "runtime@sha256:aaa", "adot": "adot@sha256:bbb"},
        },
    )
    return dict(state.value["task_input_sha256"])


def test_monitoring_install_reruns_when_a_dashboard_or_the_grafana_options_change(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    dashboards = root / "deploy/observability/dashboards"
    dashboards.mkdir(parents=True)
    (dashboards / "gpu-fault-overview.json").write_text(
        '{"uid": "a"}', encoding="utf-8"
    )

    baseline = _bind(tmp_path, root)
    unchanged = _bind(tmp_path, root)
    (dashboards / "gpu-fault-overview.json").write_text(
        '{"uid": "a", "v": 2}', encoding="utf-8"
    )
    edited = _bind(tmp_path, root)
    viewer = _bind(tmp_path, root, request_overrides={"grafana_viewer": "u-42"})
    pinned = _bind(
        tmp_path, root, request_overrides={"grafana_workspace_id": "g-5b81a13d97"}
    )

    assert baseline["monitoring_install"] == unchanged["monitoring_install"]
    assert baseline["monitoring_install"] != edited["monitoring_install"], (
        "monitoring_install ignores a changed dashboard"
    )
    assert edited["monitoring_install"] != viewer["monitoring_install"], (
        "monitoring_install ignores a new --grafana-viewer"
    )
    assert edited["monitoring_install"] != pinned["monitoring_install"], (
        "monitoring_install ignores a new --grafana-workspace-id"
    )
    for name in ("aurora_refresh", "node_keys:gpu-a", "monitoring_resources"):
        assert baseline[name] == edited[name], f"{name} re-runs on a dashboard edit"


# --- site.yaml ------------------------------------------------------------------------


def _site_with_health(tmp_path: Path, **health: Any) -> Path:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["health"].update(health)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_site_health_persists_the_grafana_workspace_and_switch(tmp_path: Path) -> None:
    rendered = load_site(
        _site_with_health(
            tmp_path, grafanaWorkspaceId="g-5b81a13d97", grafanaEnabled=True
        )
    )
    health = rendered.release_config["health"]

    assert health["grafana_workspace_id"] == "g-5b81a13d97"
    assert health["grafana_enabled"] is True
    assert rendered.release_config["health"]["amp_workspace_id"] == "ws-test"


def test_site_health_defaults_grafana_to_enabled_without_a_workspace(
    tmp_path: Path,
) -> None:
    health = load_site(site_file(tmp_path / "default")).release_config["health"]

    assert health["grafana_workspace_id"] is None
    assert health["grafana_enabled"] is True

    disabled = load_site(_site_with_health(tmp_path / "off", grafanaEnabled=False))
    assert disabled.release_config["health"]["grafana_enabled"] is False
    with pytest.raises(SiteConfigError, match="grafanaWorkspaceId"):
        load_site(_site_with_health(tmp_path / "blank", grafanaWorkspaceId=""))


# --- installation registry -------------------------------------------------------------


def test_registry_carries_the_grafana_workspace_from_bootstrap_state(
    tmp_path: Path,
) -> None:
    site = load_site(site_file(tmp_path))
    state = {
        "site_id": "test-site",
        "resources": {
            "monitoring_install": {
                "role_arn": "arn:aws:iam::123456789012:role/gpu-fault-amp-writer",
                "grafana": {
                    "status": "PROVISIONED",
                    "workspace_id": "g-5b81a13d97",
                    "ownership": "EXTERNAL",
                    "endpoint": "g-5b81a13d97.grafana-workspace.us-east-1.amazonaws.com",
                    "service_account_id": "9",
                    "dashboards": [{"uid": "a", "title": "A", "version": 1}],
                },
            }
        },
    }

    snapshot = build_installation_snapshot(site, state, {})
    by_key = {resource.resource_key: resource for resource in snapshot.resources}

    workspace = by_key["aws/grafana/workspace"]
    assert workspace.resource_id == "g-5b81a13d97"
    assert workspace.ownership is InstallationResourceOwnership.EXTERNAL
    assert workspace.delete_policy is InstallationResourceDeletePolicy.PRESERVE
    assert workspace.region == "us-east-1"
    assert by_key["aws/grafana/service-account"].dependencies == [
        "aws/grafana/workspace"
    ]


def test_a_workspace_the_deploy_created_is_registered_for_uninstall_to_delete(
    tmp_path: Path,
) -> None:
    """First deploy created the workspace, so ``uninstall --cpu-cluster delete``
    must remove it: the record is CREATED/DELETE and its role goes after it."""

    site = load_site(site_file(tmp_path))
    state = {
        "site_id": "test-site",
        "resources": {
            "monitoring_install": {
                "grafana": {
                    "status": "PROVISIONED",
                    "workspace_id": "g-created01",
                    "ownership": "CREATED",
                    "role_name": "gpu-fault-test-site-grafana",
                    "role_arn": "arn:aws:iam::123456789012:role/gpu-fault-test-site-grafana",
                    "role_ownership": "CREATED",
                    "service_account_id": "7",
                }
            }
        },
    }

    snapshot = build_installation_snapshot(site, state, {})
    by_key = {resource.resource_key: resource for resource in snapshot.resources}

    workspace = by_key["aws/grafana/workspace"]
    assert workspace.ownership is InstallationResourceOwnership.CREATED
    assert workspace.delete_policy is InstallationResourceDeletePolicy.DELETE
    assert workspace.dependencies == ["aws/grafana/workspace-role"]
    assert by_key["aws/grafana/workspace-role"].delete_policy is (
        InstallationResourceDeletePolicy.DELETE
    )


# --- CLI -----------------------------------------------------------------------------


def _deploy_arguments(*extra: str) -> list[str]:
    return [
        "deploy",
        "--cpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        "--gpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
        "--admin-email",
        "ops@example.com",
        *extra,
    ]


def test_public_deploy_carries_the_grafana_options_to_the_source_preparer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.delenv(admin_cli.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.delenv(GRAFANA_WORKSPACE_ID_ENV, raising=False)
    monkeypatch.delenv(GRAFANA_VIEWER_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    base = _deploy_arguments("--state-dir", str(tmp_path / "state"))

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    assert calls[-1]["extra_environment"] == {}, "no option, no variable"
    assert (
        admin_cli.run(
            admin_cli.parser().parse_args(
                [
                    *base,
                    "--grafana-workspace-id",
                    "g-5b81a13d97",
                    "--grafana-viewer",
                    "u-42",
                ]
            )
        )
        == 0
    )
    assert calls[-1]["extra_environment"] == {
        GRAFANA_WORKSPACE_ID_ENV: "g-5b81a13d97",
        GRAFANA_VIEWER_ENV: "u-42",
    }


def test_deploy_help_documents_the_grafana_options_and_the_mode_flag_is_gone(
    capsys,
) -> None:
    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(["deploy", "--help"])

    help_text = capsys.readouterr().out
    assert "--grafana-workspace-id WORKSPACE_ID" in help_text
    assert "--grafana-viewer SSO_USER_ID" in help_text
    assert "--grafana {" not in help_text, "the enabled/disabled/create mode is back"
    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(_deploy_arguments("--grafana", "disabled"))


def test_the_inner_deploy_builds_the_bootstrap_request_from_the_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    requests: list[BootstrapRequest] = []
    monkeypatch.setattr(
        admin_cli,
        "bootstrap_from_arns",
        lambda request: (
            requests.append(request)
            or BootstrapResult(
                site_file=path, state_file=request.state_dir / "bootstrap-state.json"
            )
        ),
    )
    monkeypatch.setattr(
        admin_cli.subprocess,
        "run",
        lambda arguments, **_k: subprocess.CompletedProcess(arguments, 0),
    )
    # The release that follows the bootstrap is not under test here.
    monkeypatch.setattr(admin_cli, "_run_automatic_release", lambda **_k: 0)
    monkeypatch.delenv(GRAFANA_WORKSPACE_ID_ENV, raising=False)
    monkeypatch.delenv(GRAFANA_VIEWER_ENV, raising=False)
    arguments = argparse.Namespace(
        command="deploy",
        file=None,
        cpu_cluster_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/cpu",
        gpu_cluster_arn=["arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a"],
        repo_root=tmp_path / "repo",
        state_dir=tmp_path / "state",
        alert_email="operations@example.com",
        staging_only_release=True,
        impact_base="origin/release",
        prepared_source_release=True,
        show_effective_config=False,
        grafana_workspace_id="g-5b81a13d97",
        grafana_viewer="u-42",
    )

    assert admin_cli.run(arguments) == 0
    (request,) = requests
    assert request.grafana_workspace_id == "g-5b81a13d97"
    assert request.grafana_viewer == "u-42"
